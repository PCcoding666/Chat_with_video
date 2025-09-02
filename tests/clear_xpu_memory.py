#!/usr/bin/env python3
"""
清理Intel XPU显存
强制释放所有占用的显存资源
"""

import os
import torch
import gc

# 设置环境变量
os.environ['DISABLE_TRITON'] = '1'
os.environ['USE_TRITON'] = '0'
os.environ['TRITON_DISABLE'] = '1'
os.environ['DISABLE_FLASH_ATTN'] = '1'
os.environ['PYTORCH_DISABLE_TRITON'] = '1'

def clear_all_xpu_memory():
    """清理所有XPU显存"""
    try:
        print("🧹 开始清理Intel XPU显存...")
        
        # 检查XPU可用性
        if not torch.xpu.is_available():
            print("❌ Intel XPU不可用")
            return False
        
        # 显示清理前的显存使用情况
        if hasattr(torch.xpu, 'memory_allocated'):
            before_alloc = torch.xpu.memory_allocated() / 1024**3
            before_reserved = torch.xpu.memory_reserved() / 1024**3 if hasattr(torch.xpu, 'memory_reserved') else 0
            print(f"📊 清理前显存使用: {before_alloc:.2f} GB (预留: {before_reserved:.2f} GB)")
        
        # 强制垃圾回收
        print("🗑️ 执行垃圾回收...")
        gc.collect()
        
        # 清理XPU缓存
        if hasattr(torch.xpu, 'empty_cache'):
            print("🧽 清理XPU缓存...")
            torch.xpu.empty_cache()
        
        # 同步XPU设备
        if hasattr(torch.xpu, 'synchronize'):
            print("🔄 同步XPU设备...")
            torch.xpu.synchronize()
        
        # 再次垃圾回收
        gc.collect()
        
        # 显示清理后的显存使用情况
        if hasattr(torch.xpu, 'memory_allocated'):
            after_alloc = torch.xpu.memory_allocated() / 1024**3
            after_reserved = torch.xpu.memory_reserved() / 1024**3 if hasattr(torch.xpu, 'memory_reserved') else 0
            
            print(f"📊 清理后显存使用: {after_alloc:.2f} GB (预留: {after_reserved:.2f} GB)")
            
            freed_memory = before_alloc - after_alloc
            if freed_memory > 0:
                print(f"✅ 成功释放显存: {freed_memory:.2f} GB")
            else:
                print("ℹ️ 没有显存需要释放")
        
        print("🎉 XPU显存清理完成!")
        return True
        
    except Exception as e:
        print(f"❌ XPU显存清理失败: {e}")
        return False

def main():
    """主函数"""
    print("="*50)
    print("🧹 Intel XPU 显存清理工具")
    print("="*50)
    
    success = clear_all_xpu_memory()
    
    if success:
        print("\n✅ 显存清理成功，可以重新运行应用")
    else:
        print("\n❌ 显存清理失败")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)