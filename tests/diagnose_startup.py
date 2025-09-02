#!/usr/bin/env python3
"""
启动诊断脚本
用于检查服务启动失败的具体原因
"""

import os
import sys
import traceback

# 设置环境变量
os.environ['DISABLE_TRITON'] = '1'
os.environ['USE_TRITON'] = '0'
os.environ['TRITON_DISABLE'] = '1'
os.environ['DISABLE_FLASH_ATTN'] = '1'
os.environ['PYTORCH_DISABLE_TRITON'] = '1'

def check_python_env():
    """检查Python环境"""
    print("🐍 Python环境检查:")
    print(f"  - Python版本: {sys.version}")
    print(f"  - Python路径: {sys.executable}")
    print(f"  - 工作目录: {os.getcwd()}")
    
def check_basic_imports():
    """检查基础导入"""
    print("\n📦 基础依赖检查:")
    
    basic_packages = [
        'torch', 'torchvision', 'transformers', 
        'gradio', 'PIL', 'numpy', 'decord'
    ]
    
    for package in basic_packages:
        try:
            if package == 'PIL':
                import PIL
                print(f"  ✅ {package}: {PIL.__version__}")
            elif package == 'gradio':
                import gradio as gr
                print(f"  ✅ {package}: {gr.__version__}")
            elif package == 'torch':
                import torch
                print(f"  ✅ {package}: {torch.__version__}")
            elif package == 'transformers':
                import transformers
                print(f"  ✅ {package}: {transformers.__version__}")
            else:
                __import__(package)
                print(f"  ✅ {package}: 导入成功")
        except ImportError as e:
            print(f"  ❌ {package}: 导入失败 - {e}")
            return False
    
    return True

def check_xpu_availability():
    """检查XPU可用性"""
    print("\n🔧 Intel XPU检查:")
    
    try:
        import torch
        
        if torch.xpu.is_available():
            print(f"  ✅ XPU可用")
            print(f"  ✅ XPU设备数量: {torch.xpu.device_count()}")
            
            # 测试XPU基本操作
            try:
                device = torch.device('xpu')
                x = torch.randn(2, 3).to(device)
                print(f"  ✅ XPU张量操作测试成功")
                return True
            except Exception as e:
                print(f"  ❌ XPU张量操作失败: {e}")
                return False
        else:
            print("  ❌ XPU不可用")
            return False
            
    except Exception as e:
        print(f"  ❌ XPU检查失败: {e}")
        return False

def check_project_imports():
    """检查项目模块导入"""
    print("\n🏗️ 项目模块检查:")
    
    try:
        # 添加项目路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # 检查核心模块
        modules_to_check = [
            'src.chat_with_video.video_encoder',
            'src.chat_with_video.model_loader', 
            'src.chat_with_video.gradio_app',
            'src.chat_with_video.video_chat_service'
        ]
        
        for module in modules_to_check:
            try:
                __import__(module)
                print(f"  ✅ {module}: 导入成功")
            except ImportError as e:
                print(f"  ❌ {module}: 导入失败 - {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ 项目模块检查失败: {e}")
        return False

def test_gradio_startup():
    """测试Gradio基础启动"""
    print("\n🌐 Gradio启动测试:")
    
    try:
        import gradio as gr
        
        # 创建简单的测试界面
        def test_function():
            return "测试成功！"
        
        with gr.Blocks() as demo:
            gr.Markdown("# 简单测试界面")
            test_btn = gr.Button("测试")
            output = gr.Textbox(label="输出")
            test_btn.click(test_function, outputs=output)
        
        print("  ✅ Gradio界面创建成功")
        
        # 尝试启动（但不阻塞）
        try:
            import threading
            import time
            
            def launch_demo():
                demo.launch(
                    server_name="127.0.0.1",
                    server_port=7862,  # 使用不同端口避免冲突
                    prevent_thread_lock=True,
                    quiet=True
                )
            
            # 启动测试
            thread = threading.Thread(target=launch_demo)
            thread.daemon = True
            thread.start()
            
            time.sleep(2)  # 等待启动
            
            print("  ✅ Gradio启动测试成功")
            return True
            
        except Exception as e:
            print(f"  ❌ Gradio启动失败: {e}")
            return False
            
    except Exception as e:
        print(f"  ❌ Gradio测试失败: {e}")
        return False

def main():
    """主诊断函数"""
    print("="*60)
    print("🩺 Chat_with_video 启动诊断")
    print("="*60)
    
    # 1. 检查Python环境
    check_python_env()
    
    # 2. 检查基础依赖
    if not check_basic_imports():
        print("\n❌ 基础依赖检查失败，请检查环境配置")
        return 1
    
    # 3. 检查XPU
    xpu_ok = check_xpu_availability()
    if not xpu_ok:
        print("\n⚠️ XPU不可用，将使用CPU模式")
    
    # 4. 检查项目模块
    if not check_project_imports():
        print("\n❌ 项目模块导入失败")
        return 1
    
    # 5. 测试Gradio
    if not test_gradio_startup():
        print("\n❌ Gradio启动测试失败")
        return 1
    
    print("\n" + "="*60)
    print("✅ 所有检查通过！环境准备就绪")
    print("="*60)
    
    # 现在尝试启动实际应用
    print("\n🚀 尝试启动实际应用...")
    
    try:
        from src.chat_with_video.gradio_app import VideoChatGradioApp
        
        app = VideoChatGradioApp()
        print("✅ 应用实例创建成功")
        
        print("🌐 启动Web界面...")
        app.launch(
            server_name="0.0.0.0",
            server_port=7861,
            share=False
        )
        
    except Exception as e:
        print(f"❌ 应用启动失败: {e}")
        print("\n详细错误信息:")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)