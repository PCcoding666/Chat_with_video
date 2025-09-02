#!/usr/bin/env python3
"""
最终XPU加载测试
验证完整的模型加载流程是否能绕过显存查询问题
"""

import os
import sys
import time

# 设置环境变量
os.environ['DISABLE_TRITON'] = '1'
os.environ['USE_TRITON'] = '0'
os.environ['TRITON_DISABLE'] = '1'
os.environ['DISABLE_FLASH_ATTN'] = '1'
os.environ['PYTORCH_DISABLE_TRITON'] = '1'

def main():
    print("🚀 最终XPU加载测试")
    print("="*40)
    
    try:
        # 添加项目路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # 导入
        print("1. 导入模块...")
        from src.chat_with_video.model_loader import MiniCPMVInference
        
        print("✅ 模块导入成功")
        
        # 创建实例
        print("\n2. 创建推理引擎...")
        engine = MiniCPMVInference(
            model_path='openbmb/MiniCPM-V-4_5-int4',
            device='xpu'
        )
        print("✅ 推理引擎创建成功")
        
        # 测试完整的初始化流程（这会触发我们的终极XPU加载策略）
        print("\n3. 测试完整初始化流程...")
        print("   这将验证是否能绕过Intel XPU显存查询问题")
        
        start_time = time.time()
        try:
            engine.initialize()
            init_time = time.time() - start_time
            print(f"✅ 模型初始化成功！耗时: {init_time:.2f}秒")
            
            # 验证模型确实在XPU上
            print("\n4. 验证模型设备...")
            device_info = engine.get_device_info()
            print("设备信息:")
            for k, v in device_info.items():
                print(f"   {k}: {v}")
            
            if device_info.get('device') == 'xpu':
                print("✅ 模型确实在XPU上运行")
                print("🎉 Intel XPU显存查询问题已彻底解决！")
            else:
                print(f"⚠️ 模型在 {device_info.get('device')} 上运行，不是XPU")
                
            # 显示模型参数分布
            print("\n5. 模型参数分布:")
            first_param = next(engine.model.parameters())
            print(f"   第一个参数设备: {first_param.device}")
            print(f"   参数总数: {sum(p.numel() for p in engine.model.parameters()):,}")
            
            return 0
            
        except Exception as init_error:
            print(f"❌ 模型初始化失败: {init_error}")
            # 检查是否是显存查询错误
            error_str = str(init_error)
            if "doesn't support querying the available free memory" in error_str:
                print("❌ 显存查询问题仍未解决！")
                return 1
            else:
                print("❌ 其他初始化错误")
                import traceback
                traceback.print_exc()
                return 1
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    input("\n按Enter键退出...")
    sys.exit(exit_code)