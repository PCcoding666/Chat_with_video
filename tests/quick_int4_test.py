#!/usr/bin/env python3
"""
快速测试INT4模型
"""
import os
import torch

# 设置环境变量
os.environ['DISABLE_TRITON'] = '1'
os.environ['USE_TRITON'] = '0'

def test_int4_model():
    try:
        print("🧪 快速测试INT4模型...")
        
        from src.chat_with_video.model_loader import MiniCPMVInference
        
        # 创建推理引擎
        print("📦 创建INT4推理引擎...")
        inference_engine = MiniCPMVInference()
        
        # 初始化模型
        print("🚀 初始化模型...")
        inference_engine.initialize()
        
        print("✅ INT4模型加载成功!")
        
        # 简单推理测试
        print("🧪 测试推理...")
        test_msgs = [{'role': 'user', 'content': ['你好']}]
        
        response = inference_engine.chat(
            msgs=test_msgs,
            max_new_tokens=20,
            temperature=0.7
        )
        
        print(f"✅ 推理成功! 回答: {response}")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_int4_model()
    print(f"\n结果: {'✅ 成功' if success else '❌ 失败'}")